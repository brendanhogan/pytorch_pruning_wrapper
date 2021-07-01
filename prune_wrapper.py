import os
import torch
import numpy as np
import torch.nn as nn



class PruningWrapper(nn.Module):
    """
    Wrapper for PyTorch model, that implements global pruning method. Pruning
    criteria can be random or magnitude pruning. Additionally allows for reset
    of parameters to random initialization or to same starting/late reseting
    initialization i.e. the case for Lottery Ticket Hypothesis experiments.

    ...

    Attributes
    ----------
    net : nn.Module
        the original module

    Methods
    -------
    set_initial_weights()
        Checkpoints current model weights as weights to be used when reinitalize()
        is called -- unless random_initialization=True is called in constructor,
        in which case this method does nothing.
    calculate_percent_pruned(verbose=False)
        Calculates the number of pruneable parameters and the number of
        parameters that have been pruned, prints out numbers if verbose=True.
    prune_weights()
        Globaly prunes the set percent of remaining weights in the model, either
        by using weight magnitude or randomly if random_prune=True called in
        constructor.
    reinitialize()
        Reverts model back to weights when set_initial_weights() was called, or
        to random weights if random_initialization=True is called in constructor.
        Either way the pruned weights do not change.

    """


    def __init__(self,network,prune_rate,device,save_dir,random_initialization=False,random_prune=False,filter_function=lambda x: True):
        """
        Parameters
        ----------
        network : nn.Module
            The PyTorch model to be wrapped
        prune_rate : float
            The percentage of remaining (unpruned) weights to be pruned each time
            prune_weights() is called. I.e. prune_rate=.2 is 20 percent remaining
            weights pruned each prune_weights() call.
        device : str
            Pytorch device to use -- should be same device network and data is
            on (cuda/cpu).
        save_dir : str
            Directory where wrapper can store weights if necessary.
        random_initialization : bool, optional
            When reinitialize() is called model can either revert to previous
            state, as set by set_initial_weights() or to random initialization
            if random_initialization is set to True (default is False).
        random_prune : bool, optional
            When prune_weights() is called the weights can either be pruned by
            weight magnitude or randomly if random_prune is True
            (default is False).
        filter_function : function, optional
            When generating mask, this class will iterate through all parameters
            in net.named_parameters(), so paremeters will only be considered
            pruneable if filter_function(name) for name in net.named_parameters()
            i.e. can filter out certain layers (like final linear layers) from being
            able to be pruned (default is all layers are pruneable).

        """
        super(PruningWrapper, self).__init__()
        # Store base network, prune rate, the path to the initial weights and device
        self.net = network
        self._prune_rate = prune_rate
        self._path_to_weights = None
        self._device = device
        self._random_initialization = random_initialization
        self._random_prune = random_prune
        self._save_dir = save_dir
        self._filter_function = filter_function
        self._mask = {} # Holds mask to be used for all weights

        # Generate initial mask
        self._generate_initial_mask()


    def _generate_initial_mask(self):
        """
        Generates initial mask object which is a dictionary mapping a unique
        idnetifier for prunable layer -> integer where 1 is on (unpruned) 0 is
        off (pruned).

        Uses the filter_function to selectively remove layers from being
        considered prunable.
        """
        self._mask = {}
        for name,param in self.net.named_parameters():
            if self._filter_function(name):
                self._mask[name] = torch.ones(list(param.shape)).to(self._device)

    def set_initial_weights(self):
        """
        Saves current model state so that at later point model can be
        re-initialized to this state. This idea was introduced by the Lottery
        Ticket Hypothesis.

        Can be used immediately after wrapping module, or after some training
        rounds (late resetting) which shows better results for larger networks.

        Usually is only called once -- but could do multiple state resets.
        """
        if not self._random_initialization:
            self._path_to_weights = os.path.join(self._save_dir,"intial_weights.pth")
            torch.save(self.net.state_dict(),self._path_to_weights)

    def _randomly_prune(self,weights,number_to_prune):
        """
        Generates new mask with next round of weights pruned randomly.
        """
        # Randomly change number_to_prune 1's in mask to 0's
        # If randomly prune, need to track position of masked weights
        # 1. Generate list and dict to track currently unmasked weights
        w_vs = []
        w_vs_loc = {}
        for k,v in weights.items():
            idx_vals =np.where(self._mask[k].cpu().detach().numpy() == 1)
            w_vs_loc[k] = idx_vals
            w_vs.append(v[self._mask[k].cpu().detach().numpy() == 1])
        w_vs_lengths = [x.shape[0] for x in w_vs]
        for i in range(1,len(w_vs_lengths)):
            w_vs_lengths[i] += w_vs_lengths[i-1]
        weight_vector = np.concatenate(w_vs)
        weight_keys = list(weights.keys())

        # 2. Randomly select indices of weight_vector to prune
        idxs = np.arange(weight_vector.shape[0])
        np.random.shuffle(idxs)
        idxs_to_prune = idxs[:number_to_prune]
        # 3. Now sort sort so can iterate through and prune
        idxs_to_prune = np.sort(idxs_to_prune)
        # 4. Go through and prune in mask
        for idx in idxs_to_prune:
            # 4a. Get layer
            layer_id = np.digitize(idx,w_vs_lengths)
            layer_key = weight_keys[layer_id]
            # 4b. Get idx relative to layer
            weight_id = idx
            if layer_id > 0:
                weight_id = weight_id - w_vs_lengths[layer_id-1]
            # 4c. Get mask idx
            loc = w_vs_loc[layer_key]
            # 4d. Get value to change
            loc_to_change = tuple([x[weight_id] for x in loc])
            # 4e. Now change value
            self._mask[layer_key][loc_to_change] = 0

    def _magnitude_prune(self,weights,number_to_prune):
        """
        Generates new mask with next round of weights pruned based off of their
        current magnitude (smallest magnitude weights pruned).
        """
        # 1. Make vector out of weights that are currently unmasked
        weight_vector = np.concatenate([v[self._mask[k].cpu().detach().numpy() == 1] for k, v in weights.items()])

        # 2. Sort weights, and identify threshold to do cutoff
        threshold = np.sort(np.abs(weight_vector))[number_to_prune]
        # 3. Now make new mask, where those weights below threshold are masked out
        new_mask = {}
        for k, v in weights.items():
            masked_values = torch.zeros(v.shape)
            masked_values[np.where(np.abs(v)>threshold)] = 1
            new_mask[k] = masked_values.to(self._device)
        # 4. Update mask
        self._mask = new_mask

    def prune_weights(self):
        """
        Prunes prune_rate percent of remaining prunable weights, either randomly
        or by weight magnitude as described in constructor.
        """
        self._apply_mask()
        # 1. First calculate number of weights to prune
        total_unpruned, _, _ = self.calculate_percent_pruned()
        number_to_prune = int(total_unpruned*self._prune_rate)

        # 2. First isolate weights that are from pruneable layers
        weights = {k: v.clone().cpu().detach().numpy() for k, v in self.net.state_dict().items() if k in self._mask.keys()}

        # 3. Now either randomly prune, or magnitude prune
        if self._random_prune:
            self._randomly_prune(weights,number_to_prune)
        else:
            self._magnitude_prune(weights,number_to_prune)



    def reinitialize(self):
        """
        Either loads state from when set_initial_weights() was called, or
        randomly reinitializes model if random_initialization
        """
        if self._random_initialization:
            for layer in self.net.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        else:
            if self._path_to_weights is None:
                print("Original weights have not yet been set -- call method set_initial_weights()")
            else:
                # Reset original weights
                self.net.load_state_dict(torch.load(self._path_to_weights))

    def _apply_mask(self):
        """
        Sets pruned weights to 0 each forward pass.

        Note: Accumulates no gradient so mask values never change.
        """
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                # Check if is pruneable layer
                if name in self._mask.keys():
                    # Then apply mask
                    param.data = param.data*self._mask[name]

    def forward(self,x):
        """
        Normal nn.Module method -- masks out pruned weights before doing normal
        forward pass.
        """
        # First apply mask, by zeroing out respective weights
        self._apply_mask()
        # Then do normal forward pass
        return self.net(x)

    def calculate_percent_pruned(self,verbose=False):
        """
        Calculates from masks, the total number of parameters, how many have
        been pruned, returns both and as percentage.
        """
        total_unpruned = 0
        total_prunable = 0
        total_number_of_parameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        for k,v in self._mask.items():
            total_unpruned += torch.sum(v)
            total_prunable += torch.sum(torch.ones_like(v))
        percent_pruned = 1. - float(total_unpruned)/total_prunable
        percent_pruned_to_total = 1. - float(total_unpruned)/total_number_of_parameters
        if verbose:
            print("Percent of model pruned: {}".format(percent_pruned.item()))
            print("Percent of parameters that are pruneable {}".format(float(total_prunable)/total_number_of_parameters))
        return total_unpruned.item(), total_prunable.item(), percent_pruned.item()














#
